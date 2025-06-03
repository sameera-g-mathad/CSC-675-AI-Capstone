import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React from "react";
import { AntDesign } from "@expo/vector-icons";
import { router } from 'expo-router'
import uuid from 'react-native-uuid'
interface tabHeaderTitle {
    addButton: boolean,
    backgroundColor: string,
    title: string;

}

const generateUUID = () => {
    return uuid.v4()
}

export default function TabHeader({ title, addButton, backgroundColor }: tabHeaderTitle) {
    return (
        <View style={{ ...styles.viewStyle, backgroundColor }}>
            <Text style={styles.textStyle}>{title}</Text>
            {addButton && <TouchableOpacity onPress={() => {
                const uuid = generateUUID();
                router.push({ pathname: `./chats/${uuid}`, params: { title: 'New Chat' } })
            }}>
                <AntDesign name='plus' size={30} />
            </TouchableOpacity>}
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyle: {
        height: 70,
        justifyContent: "space-between",
        alignItems: 'center',
        padding: 10,
        display: "flex",
        flexDirection: 'row',
        borderBottomLeftRadius: 10,
        borderBottomRightRadius: 10
    },
    textStyle: {
        fontWeight: 'bold',
        fontSize: 20,
        textTransform: 'capitalize'
    }
});
