import { StyleSheet, FlatList, Text, Pressable, View } from "react-native";
import React from "react";
import { router } from 'expo-router'


const data = [
    { id: '1', name: 'Alice' },
    { id: '2', name: 'Bob' },
    { id: '3', name: 'Charlie' },
    { id: '4', name: 'David' },
    { id: '5', name: 'Alice' },
    { id: '6', name: 'Bob' },
    { id: '7', name: 'Charlie' },
    { id: '8', name: 'David' },
    { id: '9', name: 'Alice' },
    { id: '10', name: 'Bob' },
    { id: '11', name: 'Charlie' },
    { id: '12', name: 'David' },
];

export default function chat() {
    return (
        <View style={styles.viewStyle}>
            <FlatList data={data} horizontal={false} keyExtractor={item => item.id} renderItem={({ item }) =>
                <Pressable style={styles.pressableStyle} onPress={() => {
                    router.push({
                        pathname: `./chats/${item.id}`,
                        params: { name: item.name }
                    });
                }}>
                    <Text>{item.name}</Text>
                </Pressable>
            } />
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyle: {
        flex: 1,
    },
    pressableStyle: {
        backgroundColor: 'white',
        padding: 20,
        margin: 5,
        borderRadius: '3%'
    }
});
