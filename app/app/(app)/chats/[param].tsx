import { KeyboardAvoidingView, Platform, SafeAreaView, StyleSheet, Text, View } from "react-native";
import React, { useEffect, useContext } from "react";
import { useNavigation, useLocalSearchParams, useRouter } from "expo-router";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chatbox from "@/app/_components/Chatbox";
import Content from "@/app/_components/Content";
import { HeaderBackButton } from '@react-navigation/elements';

export default function Chat() {
    const navigation = useNavigation();
    const router = useRouter()
    const { chat_title } = useContext(ChatHistContext)
    useEffect(() => {
        navigation.setOptions({
            header: () =>
            (
                <View style={{
                    height: 70,
                    padding: 10,
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    backgroundColor: '#6ee7b7'
                }}>
                    <HeaderBackButton onPress={() => router.replace('/')} />
                    < Text >{chat_title}</ Text>
                </View>
            )
        })
    }, [chat_title])
    return (
        <SafeAreaView style={styles.viewContainer}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={130}
                style={{ flex: 1 }}>
                <View style={styles.chatContainer}>
                    <Content />
                </View>
                <View>
                    <Chatbox />
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        flex: 1,
    },
    chatContainer: {
        flex: 1,
    },

});
